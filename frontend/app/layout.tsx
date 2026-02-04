import './globals.css';
import { Inter } from 'next/font/google';
import React from 'react';
import SiteHeader from '../components/layout/SiteHeader';
import SiteFooter from '../components/layout/SiteFooter';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'Research Paper Summarizer',
  description: 'A tool to analyze and summarize research papers using LLMs',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-neutral-50 text-neutral-900 bg-app`}>
        <div className="min-h-dvh flex flex-col">
          <SiteHeader />
          <main className="flex-1">{children}</main>
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
